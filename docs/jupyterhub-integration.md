# JupyterHub integration

The recommended runtime for these notebooks is the `jupyterhub` service in the [`genai-vanilla`](https://github.com/thekaveh/genai-vanilla) stack. As of genai-vanilla `448333d` (pinned in this repo's `vendor/genai-vanilla` submodule), that image natively ships the ml-eng-lab dependency set:

- `thekaveh-nnx[lm]==0.2.0`
- `python-louvain`, `nltk`, `spacy`, `torchao`, `prettytable`
- The `en_core_web_sm` spaCy model + the `vader_lexicon` NLTK corpus, downloaded at image-build time

For most workflows you do NOT need this repo's wrapper script or override file — just start the standalone stack and connect from VS Code.

## 1. Default path: standalone genai-vanilla + VS Code Mode 2

This is the recommended path for **most tier-covered ml-eng-lab notebooks** — the exception being the from-scratch `notebooks/image_classification-mnist-ffnn-numpy/notebook.ipynb` (which imports sibling `.py` modules from its own folder, requiring filesystem access, and needs the §2 path). The quantization notebook is still manual-only under `torch>=2.5` + `torchao>=0.17`.

1. Bring the stack up from a standalone clone of genai-vanilla:

    ```bash
    cd ~/repos/genai-vanilla
    ./start.sh
    ```

2. Open any ml-eng-lab notebook locally in VS Code (it stays on your host filesystem).

3. Point VS Code at the remote kernel — see [vscode-remote-access.md Mode 2](vscode-remote-access.md#2-mode-2-connect-to-remote-jupyter-server-default).

`import nnx` resolves for tier-covered notebooks in the current image. Notebook outputs save back to the local `.ipynb` file because VS Code holds the file on the host.

What this path does NOT give you: notebook code that does `pd.read_csv("./data/foo.csv")` or `NNRun.save()` writes to the container's CWD (`/home/jovyan/`), not to your host repo. Data/run artifacts land in the `jupyterhub-data` named volume — opaque to `git status` and lost on `docker volume rm`. For most Tier-A demos that's fine (small datasets, cheap to re-download). For long-running training where you want host-side persistence, see §2.

## 2. Persistence path: wrapper script + bind-mount

Use this when you want any of:

- Datasets and `runs/` checkpoints to land on your host filesystem (visible in `git status`, survives `docker compose down -v`).
- The from-scratch `notebooks/image_classification-mnist-ffnn-numpy/notebook.ipynb` notebook to work (it imports sibling `.py` modules from its own folder).
- A development workflow where you `git commit` notebook edits + dataset downloads from inside the container.

This repo vendors a snapshot of genai-vanilla as a git submodule at [`vendor/genai-vanilla`](https://github.com/thekaveh/ml-eng-lab/tree/main/vendor/genai-vanilla) and ships a wrapper script that layers an ml-eng-lab override onto the standalone compose:

### 2.1. Clone with submodules

```bash
git clone --recurse-submodules https://github.com/thekaveh/ml-eng-lab.git
# Or, if already cloned:
git submodule update --init --recursive
```

### 2.2. Run the wrapper

```bash
scripts/start-jupyterhub.sh
```

The wrapper sets `ML_REPO_PATH` (the ml-eng-lab repo root), exports `COMPOSE_FILE` to layer [`deploy/genai-vanilla-jupyterhub.override.yml`](https://github.com/thekaveh/ml-eng-lab/blob/main/deploy/genai-vanilla-jupyterhub.override.yml) onto genai-vanilla's base compose, and execs the submodule's `./start.sh`. The override bind-mounts `${ML_REPO_PATH}:/home/jovyan/work/ml-eng-lab`, so from the running container's perspective, the repo is at `/home/jovyan/work/ml-eng-lab/`.

By default, the wrapper mounts an empty ignored directory at `/home/jovyan/.ssh`; host SSH keys are not exposed to notebook code. To opt into a read-only SSH-key mount for `git push`, set `HOST_SSH_DIR` explicitly:

```bash
HOST_SSH_DIR=/path/to/keys scripts/start-jupyterhub.sh
```

## 3. nnx development: editable install override

If you're developing `nnx` itself (editing source on your host and wanting changes to land in the running kernel without a `pip install` cycle), clone [`thekaveh/NNx`](https://github.com/thekaveh/NNx) anywhere outside the ml-eng-lab tree, then bind-mount your clone into the running container alongside ml-eng-lab (extend `deploy/genai-vanilla-jupyterhub.override.yml` with a second volume) and:

```bash
docker exec -it <project>-jupyterhub pip install -e /home/jovyan/work/NNx[lm]
```

For everyone else, the image's pre-baked `thekaveh-nnx[lm]==0.2.0` layer is what you want and this override is unnecessary.

## 4. Submodule pin / bumping

`vendor/genai-vanilla` pins a known-good commit on genai-vanilla's `main`. Standard submodule bump:

```bash
cd vendor/genai-vanilla
git fetch origin
git checkout main
git pull origin main
cd ../..
git add vendor/genai-vanilla
git commit -m "ml-eng-lab: bump genai-vanilla submodule to <new-sha>"
```

The submodule pin matters for the §2 path; the §1 path uses your standalone genai-vanilla checkout and is independent of the submodule.

## 5. Tested against

genai-vanilla `448333d3b1a530fafd76d224ee1066181de8fac4`, which includes the ml-eng-lab runtime dependency block, `thekaveh-nnx[lm]==0.2.0`, and the spaCy/NLTK asset downloads in the JupyterHub image.

## 6. Common failure modes

- **`Could not find a version that satisfies the requirement nnx-pytorch`** during `docker compose build jupyterhub` — the checkout is older than the `448333d` runtime pin. Pull current genai-vanilla `main` or update this repo's submodule with `git submodule update --init --recursive`.
- **`ModuleNotFoundError: No module named 'nnx'`** in the §1 path — the image was built from an older genai-vanilla checkout. Pull current genai-vanilla `main`, rebuild the `jupyterhub` image, and confirm `services/jupyterhub/build/requirements.txt` contains `thekaveh-nnx[lm]==0.2.0`.
- **Submodule not found at `vendor/genai-vanilla/`** — run `git submodule update --init --recursive` at the repo root.
- **`ML_REPO_PATH variable is not set`** during compose up — you ran `cd vendor/genai-vanilla && ./start.sh` directly instead of using the wrapper. Use `scripts/start-jupyterhub.sh`.
- **Relative-path reads/writes go to the wrong place** (notebook does `pd.read_csv("./data/foo.csv")` but the file is on your host) — you're on the §1 path. Switch to §2 if you want host-side persistence.
- **Stack service didn't come up** — Check `docker compose ps` from inside `vendor/genai-vanilla/` (§2) or `~/repos/genai-vanilla/` (§1).
