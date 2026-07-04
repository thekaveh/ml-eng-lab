# Notebook Reorganization Design

## 1. Summary

Move every experiment notebook into a dedicated top-level `notebooks/` tree, making each
`notebooks/<experiment>/` directory the canonical runnable home for that experiment. Active
experiments, archived CodeXGLUE experiments, per-experiment READMEs, local helper code, and
runtime artifact directories move together so notebooks remain rerunnable from their new
locations. In the same migration, rename the repository identity from `ml-eng-lab` to
`ml-eng-lab` across live documentation, tooling, container names, GitHub URLs, nbviewer URLs,
and runtime path examples.

## 2. Goals

- Put all active and archived notebooks under `notebooks/`.
- Make each notebook fully rerunnable from its new directory without relying on the old
  top-level task folder as the working directory.
- Move per-experiment documentation and task-local helper code with the notebook.
- Move existing local ignored runtime artifact directories (`data/`, `runs/`, checkpoints
  when present) under the matching new experiment directory.
- Move tracked archived CodeXGLUE support artifacts (`src/`, tracked `model/` files, and
  `.gitignore`) with their archived experiments.
- Update all tooling, tests, CI paths, verifier rules, documentation links, nbviewer links,
  and notebook markdown links to the new paths.
- Update live references from `ml-eng-lab` to `ml-eng-lab`, including GitHub repository URLs,
  nbviewer URLs, Docker image tags, Codespaces paths, container bind-mount paths, examples,
  and prose that describes the current repository.
- Leave the repository with passing verification or clearly document any environment-limited
  check that cannot be run locally.

## 3. Non-Goals

- Do not redesign the ML experiments or change model behavior.
- Do not re-execute expensive Tier-B or Tier-C notebooks as part of the structural move.
- Do not rewrite notebook outputs except where a path-only metadata or markdown update is
  required.
- Do not preserve compatibility with old notebook paths beyond updated documentation and git
  history.
- Do not rewrite historical changelog entries or preserved notebook outputs merely because
  they mention `ml-eng-lab`; only update them when the text is current guidance, a live link, or
  an active path contract.
- Do not introduce a new package layout for shared ML code; `thekaveh-nnx` remains the shared
  library dependency.

## 4. Target Layout

Active experiment directories move from the repository root to `notebooks/`:

```text
notebooks/image_classification-mnist-ffnn-numpy/
  README.md
  notebook.ipynb
  *.py
  data/        # ignored local artifact, if present

notebooks/image_classification-mnist-ffnn-numpy/
  README.md
  notebook.ipynb
  *.py
  data/        # ignored local artifact, if present
```

All other active tasks follow the same rule:

```text
<task>/
  README.md
  *.ipynb
  data/        # ignored local artifact, if present
  runs/        # ignored local artifact, if present

notebooks/<task>/
  README.md
  *.ipynb
  data/        # ignored local artifact, if present
  runs/        # ignored local artifact, if present
```

Archived CodeXGLUE experiments move under `notebooks/archive/`:

```text
notebooks/archive/codexglue_summarization/<experiment>/
  notebook.ipynb
  src/         # tracked for some experiments
  model/       # tracked result artifacts for some experiments

notebooks/archive/codexglue_summarization/<experiment>/
  notebook.ipynb
  src/
  model/
```

Top-level active task directories should be removed after their contents are moved. The
legacy `archive/README.md` should move to `notebooks/archive/README.md`, and the legacy
`archive/` directory should be removed if no other non-notebook archive material remains.

## 5. Runtime Model

Each notebook executes from its own directory. Existing simple relative paths should continue
to work after the move:

- `root="./data"` resolves to `notebooks/<experiment>/data`.
- `./runs` and NNx default run directories resolve under `notebooks/<experiment>/runs`.
- NumPy sibling imports such as `from utils import Utils` resolve because helper `.py` files
  move beside the notebook.
- Archived CodeXGLUE notebook references to local `src/` and `model/` files resolve because
  those directories move with each archived experiment.

Where notebooks link to another notebook, links must be recalculated from the new notebook
directory. For example, links from one active experiment to another now usually need
`../<other-experiment>/<notebook>.ipynb`.

## 6. Tooling Changes

Update every path contract that currently assumes notebooks live in root task directories:

- `Makefile`
  - Update `TIER_A`, `TIER_B`, and `TIER_C` notebook paths.
  - Adjust papermill loops so they still `cd` into the notebook directory and run the
    basename.
  - Keep `check-tier-a-clean` scoped to the new Tier-A paths.
- `.github/workflows/ci.yml`
  - Update Tier-A artifact upload paths.
  - Update comments that mention old notebook paths or root task folders.
- `scripts/verify_repo_config.yaml`
  - Update `active_task_dirs` to represent experiment directories under `notebooks/`.
  - Update all `required_sections` and `tier_a_notebooks` paths.
- `scripts/verify_repo.py`
  - Discover active notebooks from `notebooks/<active-task>/`.
  - Keep archived notebooks excluded from active checks while recognizing their new
    `notebooks/archive/` location.
  - Update numbered-doc scanning to inspect per-experiment READMEs under `notebooks/`.
  - Update Tier-C baseline comparison paths and any phase3 glob logic.
- `tests/`
  - Update hardcoded notebook paths and synthetic expectations.
  - Update active notebook discovery tests to account for `notebooks/archive/` exclusions.
- `pyproject.toml`
  - Update ruff per-file ignores for Tier-C notebooks.
- `.gitignore`
  - Keep broad ignored artifact patterns effective under `notebooks/**/data/`,
    `notebooks/**/runs/`, `notebooks/**/model/`, and checkpoints.

## 7. Documentation Changes

Update documentation so the new structure and repository identity are explicit and old paths
are not advertised:

- Root `README.md`
  - Title and repository name.
  - Repository layout section.
  - Quick start examples.
  - Task table links.
  - Notebook re-execution policy.
  - Archive links.
  - GitHub, Codespaces, Docker, and nbviewer examples using `ml-eng-lab`.
- `CONTRIBUTING.md`
  - Replace the old flat task-folder convention with the new `notebooks/<task>/` convention.
  - Update new-task scaffolding instructions.
  - Update nbviewer examples.
  - Update verifier and Makefile registration instructions.
  - Update current-repo prose from `ml-eng-lab` to `ml-eng-lab`.
- Per-experiment `README.md` files
  - Update links to notebooks and docs after the README moves under `notebooks/<task>/`.
  - Update nbviewer URLs to include `notebooks/`.
  - Update nbviewer URLs to use `thekaveh/ml-eng-lab`.
  - Update relative links to root docs, usually from `../docs/...` to `../../docs/...`.
- `docs/*.md`
  - Update runtime, environment, VS Code, JupyterHub, dependency, and findings references to
    old notebook paths and live `ml-eng-lab` repository paths.
- `CHANGELOG.md`
  - Add a new unreleased entry describing the notebook move and repo rename.
  - Historical entries may remain historically accurate unless they contain live guidance,
    live links, or current path contracts.
- Notebook markdown cells
  - Update links that point to old notebook paths, docs, GitHub URLs, or nbviewer URLs.

## 8. Repository Rename Handling

Treat the repo rename from `ml-eng-lab` to `ml-eng-lab` as part of the same implementation:

- Update current repository branding and prose from `ml-eng-lab` to `ml-eng-lab`.
- Update GitHub URLs from `github.com/thekaveh/ml-eng-lab` to
  `github.com/thekaveh/ml-eng-lab`.
- Update nbviewer URLs from `nbviewer.org/github/thekaveh/ml-eng-lab/...` to
  `nbviewer.org/github/thekaveh/ml-eng-lab/...`, while also inserting the new
  `notebooks/` path segment for moved notebooks.
- Update Codespaces path examples from `/workspaces/ml-eng-lab` to
  `/workspaces/ml-eng-lab`.
- Update JupyterHub and Docker bind-mount examples from `/home/jovyan/work/ml-eng-lab` to
  `/home/jovyan/work/ml-eng-lab`.
- Update Docker tags such as `ml-eng-lab` and `ml-eng-lab-ci` to `ml-eng-lab` and
  `ml-eng-lab-ci`.
- Update scripts and deploy comments or variables only when they encode user-facing
  names, paths, or container-visible mount points. Keep generic variable names such as
  `ML_REPO_PATH` if they remain semantically useful and do not expose the old repo name.
- Preserve historical mentions in old changelog entries, findings, and notebook outputs
  when they describe past events or recorded execution output rather than current usage.
- Add focused stale-reference scans that distinguish live references requiring updates from
  intentionally preserved historical references.

## 9. Data And Artifact Handling

Tracked files move through git so history is preserved as much as possible:

- notebooks
- READMEs
- task-local helper `.py` files
- archived `src/` files
- tracked archived `model/` outputs
- archive `.gitignore` files that still apply

Ignored local artifact directories move with filesystem operations, not git tracking:

- active `data/`
- active `runs/`
- root `runs/`, if it represents notebook-generated state that should live under a specific
  experiment; otherwise leave it ignored at root and document why
- checkpoints, if present

Before moving ignored artifacts, inventory them and avoid overwriting existing destination
directories. If source and destination both exist, merge only when the contents are clearly
non-conflicting; otherwise stop and report the conflict.

## 10. Verification Plan

The implementation is complete only after verification passes or a local environment
limitation is explicitly reported:

1. `python scripts/verify_repo.py --check all --fast`
2. `pytest tests/ -v`
3. `ruff check . --no-cache`
4. `make check-tier-a-clean`
5. YAML/config parse checks for workflow and verifier config
6. `bash -n scripts/start-jupyterhub.sh`
7. `git diff --check`
8. Focused stale-path scans for old active and archive notebook locations
9. Focused notebook parse checks for all moved notebooks
10. Targeted papermill smoke checks where feasible without replaying the full expensive fleet
11. Focused stale-reference scans for live `ml-eng-lab`, `thekaveh/ml-eng-lab`,
    `/workspaces/ml-eng-lab`, `/home/jovyan/work/ml-eng-lab`, and `nbviewer` references

## 11. Risks And Mitigations

- Risk: path updates miss a verifier, CI, or test contract.
  - Mitigation: run stale-path scans and full local verification.
- Risk: notebooks with `./data` or sibling imports break after moving.
  - Mitigation: move helper code and ignored local artifacts with notebooks; keep papermill
    execution rooted in the notebook directory.
- Risk: archived notebooks contain historical absolute paths in outputs.
  - Mitigation: treat historical outputs as preserved artifacts; update live links and path
    contracts, not old recorded logs unless they are current documentation.
- Risk: Tier-C baseline comparison fails after the path move.
  - Mitigation: update baseline lookup to compare new HEAD files against old baseline paths
    via an explicit mapping, or document the structural rename in the verifier logic.
- Risk: git diff is noisy because notebooks are large JSON files.
  - Mitigation: use git moves where possible and avoid unnecessary notebook rewrites.
- Risk: repository rename updates accidentally rewrite historical records.
  - Mitigation: separate live guidance/path contracts from historical changelog and
    preserved output text; update only the former.
- Risk: external links break until the remote GitHub repository is actually renamed.
  - Mitigation: make the code/docs internally consistent with the intended `ml-eng-lab`
    target and call out that remote repository rename is an external follow-up if needed.

## 12. Acceptance Criteria

- All active notebooks live under `notebooks/<task>/`.
- All archived CodeXGLUE notebooks live under `notebooks/archive/codexglue_summarization/`.
- Per-experiment READMEs and helper files live beside their notebooks.
- Top-level active task directories no longer exist.
- Notebooks are rerunnable from their new directories with relative `./data`, `./runs`,
  `./model`, and sibling imports.
- CI, Makefile, verifier config, verifier code, ruff config, tests, and docs reference the
  new paths.
- Live repository identity references use `ml-eng-lab`, including GitHub URLs, nbviewer URLs,
  Docker tags, Codespaces examples, and JupyterHub bind-mount examples.
- Stale scans find no live references to old notebook paths except historical changelog or
  preserved output contexts that are intentionally documented.
- Stale scans find no live references to `ml-eng-lab` except historical changelog, findings, or
  preserved output contexts that are intentionally documented.
- Required verification commands pass or have explicit environment-limited exceptions.
