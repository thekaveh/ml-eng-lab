# 1. Notebook Reorganization Design Record

## 1.1. Status

This document is a completed historical design record for the 2026-07-04 notebook
reorganization. It is not an active implementation checklist.

## 1.2. Decision

The repository now uses `notebooks/<experiment>/` as the canonical runnable home for every
active experiment. Archived CodeXGLUE experiments live under `notebooks/archive/`.

The live repository identity is `ml-eng-lab`. Current GitHub, nbviewer, Docker, Codespaces,
JupyterHub, and documentation-site references should use that name. Historical changelog,
maintenance, and preserved notebook-output text may retain older names only when it is clearly
recording past state.

## 1.3. Goals Satisfied

- Active notebooks live under `notebooks/<task>/`.
- Archived CodeXGLUE notebooks live under `notebooks/archive/codexglue_summarization/`.
- Per-experiment READMEs, helper modules, ignored `data/` directories, ignored `runs/`
  directories, and tracked archive support files moved with their owning notebooks.
- Tooling, tests, CI artifact paths, verifier rules, documentation links, nbviewer links, and
  notebook markdown links were updated to the new paths.
- Papermill execution continues to change into each notebook directory before execution, so
  notebook-local `./data` and `./runs` references stay local to the experiment directory.

## 1.4. Canonical Layout

```text
notebooks/
  <active-task>/
    README.md
    *.ipynb
    data/      # ignored local artifact, when present
    runs/      # ignored local artifact, when present
    *.py       # task-local helpers, when present
  archive/
    README.md
    codexglue_summarization/
      <archived-experiment>/
        notebook.ipynb
        src/    # tracked for some experiments
        model/  # tracked result artifacts for some experiments
```

## 1.5. Verification Evidence

The implementation was validated with the repository verifier, pytest, ruff, Makefile dry-runs,
stale-reference scans, notebook path scans, and git layout checks. Later maintenance passes
added a documentation-site scaffold and a verifier regression that rejects stale live guidance
describing task folders as top-level directories.
