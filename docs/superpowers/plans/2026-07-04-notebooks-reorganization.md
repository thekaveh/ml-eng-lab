# 1. Notebooks Reorganization Implementation Record

## 1.1. Status

Completed. This file is retained as a concise historical record instead of an active
checkbox plan.

## 1.2. Objective

Move every active and archived experiment into `notebooks/`, make each notebook rerunnable
from its new directory, and update live repository identity references to `ml-eng-lab`.

## 1.3. Implemented Changes

- Moved active experiment directories to `notebooks/<task>/`.
- Moved preserved CodeXGLUE experiments to `notebooks/archive/codexglue_summarization/`.
- Moved tracked archive support artifacts and local ignored runtime artifacts with their
  owning experiments.
- Preserved root-level duplicate run artifacts under task-local `runs/root-<id>` directories
  where necessary.
- Updated `Makefile`, `scripts/verify_repo.py`, `scripts/verify_repo_config.yaml`, CI paths,
  tests, README, CONTRIBUTING, runtime docs, per-task READMEs, and notebook markdown/output
  path references.
- Updated live repository identity references to `ml-eng-lab`.

## 1.4. Canonical Runtime Contract

Each notebook is executed from its own directory. Task-local relative paths therefore resolve
as follows:

- `./data` resolves to `notebooks/<task>/data`.
- `./runs` resolves to `notebooks/<task>/runs`.
- Sibling imports resolve beside the notebook, notably for the NumPy MNIST task-local helper
  modules.
- Archived notebooks are preserved for historical reference and are not part of active
  rerun guarantees.

## 1.5. Follow-Up Guardrails

- New active task folders must be created under `notebooks/`.
- `notebooks/archive/` remains read-only unless a future migration explicitly supersedes it.
- Verifier docs checks should reject live guidance that points contributors back to the old
  flat root-task layout.
- Documentation-site pages should treat `README.md` and `docs/` as the canonical source for
  current project structure.
