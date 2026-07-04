# 1. Architecture

This page describes the repository as a notebook-driven ML lab rather than as a deployable
service. The primary runtime objects are experiment directories, notebook execution tiers,
validation scripts, and documentation surfaces.

## 1.1. System Context

The diagram below is generated as a standalone HTML architecture artifact and embedded into
the documentation site.

<iframe class="architecture-frame" src="../diagrams/ml-eng-lab-system.html" title="ml-eng-lab system architecture"></iframe>

[Open the diagram in a full page](diagrams/ml-eng-lab-system.html).

## 1.2. Runtime Flow

1. A contributor opens the repository through a local venv, Docker image, Codespaces, or the
   vendored genai-vanilla JupyterHub stack.
2. They run or edit an experiment under `notebooks/<task>/`.
3. Notebook-local `./data/` and `./runs/` paths resolve inside that experiment directory.
4. `Makefile` targets execute notebooks by changing into each notebook directory before
   invoking papermill.
5. `scripts/verify_repo.py`, pytest, ruff, and CI verify structure, documentation, and public
   notebook surfaces before changes are merged.
6. MkDocs builds this documentation site from checked-in Markdown and publishes it through
   GitHub Pages.

## 1.3. Boundary Decisions

- `notebooks/archive/` is preserved as read-only historical material and excluded from active
  notebook validation.
- `thekaveh-nnx[lm]==0.2.0` is consumed from PyPI; shared library changes land upstream in
  `thekaveh/NNx` before this repo bumps the pin.
- The quantization notebook is active but manual-only until the pinned Torch stack can satisfy
  `torchao>=0.17`.
