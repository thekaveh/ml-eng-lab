# JupyterHub integration

The recommended runtime for these notebooks is the `jupyterhub` service in the [`genai-vanilla`](https://github.com/thekaveh/genai-vanilla) stack. As of genai-vanilla `cbad341` (PR #26, 2026-06-02), that image natively ships the full ml-lab dependency set:

- `nnx-pytorch` (installed as `nnx` — the PyTorch toolkit at [`thekaveh/NNx`](https://github.com/thekaveh/NNx))
- `python-louvain`, `nltk`, `spacy`, `torchao`, `prettytable`
- The `en_core_web_sm` spaCy model + the `vader_lexicon` NLTK corpus, downloaded at image-build time

For most workflows you do NOT need this repo's wrapper script, override file, or `setup-in-jupyter.sh`. Just start the standalone stack and connect from VS Code.

## 1. Default path: standalone genai-vanilla + VS Code Mode 2

This is the recommended path for **26 of the 29 ml-lab notebooks** — every Tier-A/B/C notebook except (a) the from-scratch `image_classification-mnist-ffnn-numpy/notebook.ipynb` (which imports sibling `.py` modules from its own folder, requiring filesystem access), and (b) the two notebooks that call `train_bpe`/`NNTokenizerParams` and need the `nnx[lm]` extra — `text_generation-tinyshakespeare-transformer-pytorch/notebook.ipynb` and `preference_alignment-toy-dpo-pytorch/notebook.ipynb`. The standalone genai-vanilla image currently bakes `nnx-pytorch` without extras; jumps to 28/29 once the upstream image picks up `nnx-pytorch[lm]`, tracked as a follow-up to issue #12.

1. Bring the stack up from a standalone clone of genai-vanilla:

    ```bash
    cd ~/repos/genai-vanilla
    ./start.sh
    ```

2. Open any ml-lab notebook locally in VS Code (it stays on your host filesystem).

3. Point VS Code at the remote kernel — see [vscode-remote-access.md Mode 2](vscode-remote-access.md#2-mode-2--connect-to-remote-jupyter-server-default).

`import nnx` and every other top-level import resolves out of the box. Notebook outputs save back to the local `.ipynb` file because VS Code holds the file on the host.

What this path does NOT give you: notebook code that does `pd.read_csv("./data/foo.csv")` or `NNRun.save()` writes to the container's CWD (`/home/jovyan/`), not to your host repo. Data/run artifacts land in the `jupyterhub-data` named volume — opaque to `git status` and lost on `docker volume rm`. For most Tier-A demos that's fine (small datasets, cheap to re-download). For long-running training where you want host-side persistence, see §2.

## 2. Persistence path: wrapper script + bind-mount

Use this when you want any of:

- Datasets and `runs/` checkpoints to land on your host filesystem (visible in `git status`, survives `docker compose down -v`).
- The from-scratch `image_classification-mnist-ffnn-numpy/notebook.ipynb` notebook to work (it imports sibling `.py` modules from its own folder).
- A development workflow where you `git commit` notebook edits + dataset downloads from inside the container.

This repo vendors a snapshot of genai-vanilla as a git submodule at [`vendor/genai-vanilla`](../vendor/genai-vanilla) and ships a wrapper script that layers an ml-lab override onto the standalone compose:

### 2.1 Clone with submodules

```bash
git clone --recurse-submodules https://github.com/thekaveh/ml-lab.git
# Or, if already cloned:
git submodule update --init --recursive
```

### 2.2 Run the wrapper

```bash
scripts/start-jupyterhub.sh
```

The wrapper sets `ML_REPO_PATH` (the ml-lab repo root) and `HOST_SSH_DIR` (defaults to `~/.ssh`), exports `COMPOSE_FILE` to layer [`deploy/genai-vanilla-jupyterhub.override.yml`](../deploy/genai-vanilla-jupyterhub.override.yml) onto genai-vanilla's base compose, and execs the submodule's `./start.sh`. The override bind-mounts `${ML_REPO_PATH}:/home/jovyan/work/ml-lab`, so from the running container's perspective, the repo is at `/home/jovyan/work/ml-lab/`.

To run with custom paths:

```bash
HOST_SSH_DIR=/path/to/keys scripts/start-jupyterhub.sh
```

## 3. nnx development: editable install override

If you're hacking on `nnx` itself (editing source under `nnx/src/nnx/` and wanting changes to land in the running kernel without a `pip install` cycle), the image's pip-installed `nnx-pytorch` needs to be overridden with an editable install pointing at the bind-mounted submodule.

```bash
docker exec -it <project>-jupyterhub /home/jovyan/work/ml-lab/scripts/setup-in-jupyter.sh
```

This is a no-op for the default §1 path (no bind-mount → no nnx source on disk → script errors out). It only makes sense when you're running the §2 path AND actively editing nnx.

For everyone else, the pip-installed nnx is what you want and this script is unnecessary.

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

- **`Could not find a version that satisfies the requirement nnx-pytorch`** during `docker compose build jupyterhub` — `nnx-pytorch` hasn't propagated to PyPI yet. Re-run the build after publication completes, or hold genai-vanilla at a pre-PR-26 commit and use the §2 + §3 path (wrapper + editable nnx).
- **`ModuleNotFoundError: No module named 'nnx'`** in the §1 path — the image was built before genai-vanilla PR #26 (`cbad341`). Pull the latest genai-vanilla `main` and `docker compose build jupyterhub`.
- **`ModuleNotFoundError: No module named 'nnx'`** in the §2 path — `setup-in-jupyter.sh` wasn't run for this container instance AND the image is pre-PR-26. Either upgrade genai-vanilla (preferred) or run the editable-install script.
- **Submodule not found at `vendor/genai-vanilla/`** — run `git submodule update --init --recursive` at the repo root.
- **`ML_REPO_PATH variable is not set`** during compose up — you ran `cd vendor/genai-vanilla && ./start.sh` directly instead of using the wrapper. Use `scripts/start-jupyterhub.sh`.
- **Relative-path reads/writes go to the wrong place** (notebook does `pd.read_csv("./data/foo.csv")` but the file is on your host) — you're on the §1 path. Switch to §2 if you want host-side persistence.
- **Stack service didn't come up** — Check `docker compose ps` from inside `vendor/genai-vanilla/` (§2) or `~/repos/genai-vanilla/` (§1).
