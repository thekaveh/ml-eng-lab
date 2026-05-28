# Contributing

A short guide for adding new task folders and modifying shared code in this lab.

## 1. Conventions

- This is a notebook-driven ML lab. Each top-level folder is a self-contained task (`[task]-[dataset]-[model]-[framework]`). The flat top-level layout is intentional — don't introduce a `tasks/` subdirectory or family-prefixed dirs (`vision/`, `nlp/`, …).
- Shared library code lives in the **`nnx` submodule** at `./nnx` → [`thekaveh/NNx`](https://github.com/thekaveh/NNx). Notebooks import via `from nnx.X import Y`. Do not reintroduce a local `common/` directory — `scripts/verify_repo.py` enforces this via `S7.forbidden_toplevel`.
- The `archive/` directory holds preserved-as-is experiments. Read-only.
- New notebooks should include a top markdown cell stating purpose and dataset, plus the canonical §1–§6 hierarchy (Overview / Setup / Data / Model / Training / Evaluation & Results). Phase-1 exploration notebooks use a variant: §1, §2, §3 Dataset deep-dive.

## 2. Workflow

1. Open a feature branch off `main`.
2. Make your change.
3. Run `python scripts/verify_repo.py --check all --fast` — must exit 0 (no error-severity findings; warnings are OK).
4. If you touched a notebook, re-run it (Tier-A: `make run-tier-a`; Tier-B: `make smoke-tier-b`; Tier-C: `make smoke-tier-c`). Tier-C source notebooks must remain byte-equal to the `pre-cleanup-baseline` tag — verify check E5 enforces this.
5. Open a PR. CI runs Tier-A automatically; Tier-B/C run on schedule and on `workflow_dispatch`.

## 3. Adding a new task folder

Convention: top-level folder named `[task]-[dataset]-[model]-[framework]/`.

1. Survey [`nnx/src/nnx/`](nnx/src/nnx/) for reusable primitives.
2. Identify gaps. If you need new primitives, **land them in [`thekaveh/NNx`](https://github.com/thekaveh/NNx) first** (branch in `./nnx`, commit, push), then bump the submodule pointer here.
3. Scaffold the new task folder with a `README.md` (use [`node_classification-reddit-gnn-pyg/README.md`](node_classification-reddit-gnn-pyg/README.md) as template) and notebook(s). At the top of §3 "What's in the notebook(s)", include the nbviewer tip — GitHub's notebook renderer chokes on cells with large embedded matplotlib PNGs:

   ```markdown
   > **Tip:** GitHub may show "Unable to render code block" on output cells with large matplotlib PNGs. [View this notebook on nbviewer](https://nbviewer.org/github/thekaveh/ml-lab/blob/main/<folder>/<notebook>.ipynb) for full rendering.
   ```

   For folders with multiple notebooks, link to the folder view at `https://nbviewer.org/github/thekaveh/ml-lab/tree/main/<folder>/` instead.
4. Add the notebook(s) to `required_sections` in [`scripts/verify_repo_config.yaml`](scripts/verify_repo_config.yaml) — or accept the default 6-section requirement.
5. If Tier-A, add the notebook path to `tier_a_notebooks` in the same YAML and to `TIER_A` in [`Makefile`](Makefile).
6. Update the root README's task table.
7. Tick the box on the root README roadmap.
8. YAGNI: don't add abstractions to `nnx` speculatively. Only land features when a concrete task needs them.

## 4. Modifying shared code

- **`nnx/` is a submodule.** Don't bump the submodule pointer here without a corresponding upstream commit on `thekaveh/NNx`. Workflow:
  1. `cd nnx && git checkout -b your-feature`
  2. Make your change in the submodule, add a test, commit, push.
  3. Open and merge an upstream PR.
  4. `cd nnx && git pull && cd .. && git add nnx && git commit`.
- **`vendor/genai-vanilla/` is vendored.** Don't edit it directly. The ml-specific compose override lives in [`deploy/`](deploy/) — never commit override files inside `vendor/genai-vanilla/`.
- **`archive/` is read-only.** Preserved Aug-2023 work.

## 5. Running notebooks

Primary runtime: the `genai-vanilla` stack vendored as a submodule at `vendor/genai-vanilla` (pinned to genai-vanilla's `main`).

- Start via `scripts/start-jupyterhub.sh` (NOT `cd vendor/genai-vanilla && ./start.sh` — the override needs the wrapper to set env vars and `COMPOSE_FILE`).
- Inside the running container, run `/home/jovyan/work/ml-lab/scripts/setup-in-jupyter.sh` once to pip install -e the nnx submodule.
- See [docs/jupyterhub-integration.md](docs/jupyterhub-integration.md) for the full setup.

## 6. Verification

`scripts/verify_repo.py` is the repo's four-check oracle. Run before commits / PRs:

- `python scripts/verify_repo.py --check all --fast` — structure, docs, comments, env-limited execution. Fast (<30s).
- `python scripts/verify_repo.py --check all` — adds the full Tier-A/B/C papermill smoke. Requires the genai-vanilla container or an equivalent fully-provisioned env.

Exit code 0 iff zero error-severity findings; warnings are informational. Tier-C output preservation is enforced by check E5 against the `pre-cleanup-baseline` git tag. Edits to phase3 notebooks must use `scripts/edit_notebook_markdown.py`.

## 7. One concern per PR

- Don't bundle unrelated cleanup with a feature change.
- Tier-C notebook re-execution belongs in its own PR if you ever need to (rare; preserved outputs are intentional).
