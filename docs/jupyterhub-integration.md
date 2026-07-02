# JupyterHub integration

The recommended runtime for these notebooks is the `jupyterhub` service in the [`genai-vanilla`](https://github.com/thekaveh/genai-vanilla) stack. As of genai-vanilla `cbad341` (PR #26, 2026-06-02), that image natively ships the ml-lab dependency set:

- `python-louvain`, `nltk`, `spacy`, `torchao`, `prettytable`
- The `en_core_web_sm` spaCy model + the `vader_lexicon` NLTK corpus, downloaded at image-build time
- nnx — the image's pip layer currently installs the now-defunct `nnx-pytorch[lm]` distribution name. ml-lab switched to the PyPI-stable `thekaveh-nnx[lm]==0.2.0` on 2026-06-14; the genai-vanilla image needs a coordinated upstream bump (see §6 failure mode and the 2026-06-14 entry in `CHANGELOG.md`).

For most workflows you do NOT need this repo's wrapper script or override file — just start the standalone stack and connect from VS Code.

## 1. Default path: standalone genai-vanilla + VS Code Mode 2

This is the recommended path for **most tier-covered ml-lab notebooks** — the exception being the from-scratch `image_classification-mnist-ffnn-numpy/notebook.ipynb` (which imports sibling `.py` modules from its own folder, requiring filesystem access, and needs the §2 path). The quantization notebook is still manual-only under `torch>=2.5` + `torchao>=0.17`. Until the genai-vanilla image bumps its baked `nnx-pytorch` name to `thekaveh-nnx[lm]==0.2.0`, the standalone path also requires a per-session manual fix (`docker exec ... pip install thekaveh-nnx[lm]==0.2.0` inside the running jupyterhub container) for notebooks that import `nnx`.

1. Bring the stack up from a standalone clone of genai-vanilla:

    ```bash
    cd ~/repos/genai-vanilla
    ./start.sh
    ```

2. Open any ml-lab notebook locally in VS Code (it stays on your host filesystem).

3. Point VS Code at the remote kernel — see [vscode-remote-access.md Mode 2](vscode-remote-access.md#2-mode-2--connect-to-remote-jupyter-server-default).

After the image bump, or after the per-session install workaround, `import nnx` resolves for tier-covered notebooks. Notebook outputs save back to the local `.ipynb` file because VS Code holds the file on the host.

What this path does NOT give you: notebook code that does `pd.read_csv("./data/foo.csv")` or `NNRun.save()` writes to the container's CWD (`/home/jovyan/`), not to your host repo. Data/run artifacts land in the `jupyterhub-data` named volume — opaque to `git status` and lost on `docker volume rm`. For most Tier-A demos that's fine (small datasets, cheap to re-download). For long-running training where you want host-side persistence, see §2.

## 2. Persistence path: wrapper script + bind-mount

Use this when you want any of:

- Datasets and `runs/` checkpoints to land on your host filesystem (visible in `git status`, survives `docker compose down -v`).
- The from-scratch `image_classification-mnist-ffnn-numpy/notebook.ipynb` notebook to work (it imports sibling `.py` modules from its own folder).
- A development workflow where you `git commit` notebook edits + dataset downloads from inside the container.

This repo vendors a snapshot of genai-vanilla as a git submodule at [`vendor/genai-vanilla`](../vendor/genai-vanilla) and ships a wrapper script that layers an ml-lab override onto the standalone compose:

### 2.1. Clone with submodules

```bash
git clone --recurse-submodules https://github.com/thekaveh/ml-lab.git
# Or, if already cloned:
git submodule update --init --recursive
```

### 2.2. Run the wrapper

```bash
scripts/start-jupyterhub.sh
```

The wrapper sets `ML_REPO_PATH` (the ml-lab repo root), exports `COMPOSE_FILE` to layer [`deploy/genai-vanilla-jupyterhub.override.yml`](../deploy/genai-vanilla-jupyterhub.override.yml) onto genai-vanilla's base compose, and execs the submodule's `./start.sh`. The override bind-mounts `${ML_REPO_PATH}:/home/jovyan/work/ml-lab`, so from the running container's perspective, the repo is at `/home/jovyan/work/ml-lab/`.

By default, the wrapper mounts an empty ignored directory at `/home/jovyan/.ssh`; host SSH keys are not exposed to notebook code. To opt into a read-only SSH-key mount for `git push`, set `HOST_SSH_DIR` explicitly:

```bash
HOST_SSH_DIR=/path/to/keys scripts/start-jupyterhub.sh
```

## 3. nnx development: editable install override

If you're hacking on `nnx` itself (editing source on your host and wanting changes to land in the running kernel without a `pip install` cycle), clone [`thekaveh/NNx`](https://github.com/thekaveh/NNx) anywhere outside the ml-lab tree, then bind-mount your clone into the running container alongside ml-lab (extend `deploy/genai-vanilla-jupyterhub.override.yml` with a second volume) and:

```bash
docker exec -it <project>-jupyterhub pip install -e /home/jovyan/work/NNx[lm]
```

For everyone else, the PyPI-installed `thekaveh-nnx` (or, once the image bumps, the pre-baked layer) is what you want and this override is unnecessary.

## 4. Submodule pin / bumping

`vendor/genai-vanilla` pins a known-good commit on genai-vanilla's `main`. Standard submodule bump:

```bash
cd vendor/genai-vanilla
git fetch origin
git checkout main
git pull origin main
cd ../..
git add vendor/genai-vanilla
git commit -m "ml-lab: bump genai-vanilla submodule to <new-sha>"
```

The submodule pin matters for the §2 path; the §1 path uses your standalone genai-vanilla checkout and is independent of the submodule.

## 5. Tested against

genai-vanilla `cbad341` (PR #26, 2026-06-02) or later — the first commit where the ml-lab dep set is baked into the jupyterhub image.

## 6. Common failure modes

- **`Could not find a version that satisfies the requirement nnx-pytorch`** during `docker compose build jupyterhub` — the `nnx-pytorch` PyPI name was retired on 2026-06-14 in favor of `thekaveh-nnx`. A build failure needs an upstream image bump or local image patch to install `thekaveh-nnx[lm]==0.2.0`; there is no running container to patch with `docker exec`.
- **`ModuleNotFoundError: No module named 'nnx'`** in the §1 path — the image was built before genai-vanilla PR #26 (`cbad341`) OR before the image bumps to `thekaveh-nnx[lm]`. Pull the latest genai-vanilla `main` and `docker compose build jupyterhub`; if the container already starts but only lacks the modern package, run `docker exec -it <project>-jupyterhub pip install 'thekaveh-nnx[lm]==0.2.0'` per session.
- **Submodule not found at `vendor/genai-vanilla/`** — run `git submodule update --init --recursive` at the repo root.
- **`ML_REPO_PATH variable is not set`** during compose up — you ran `cd vendor/genai-vanilla && ./start.sh` directly instead of using the wrapper. Use `scripts/start-jupyterhub.sh`.
- **Relative-path reads/writes go to the wrong place** (notebook does `pd.read_csv("./data/foo.csv")` but the file is on your host) — you're on the §1 path. Switch to §2 if you want host-side persistence.
- **Stack service didn't come up** — Check `docker compose ps` from inside `vendor/genai-vanilla/` (§2) or `~/repos/genai-vanilla/` (§1).
